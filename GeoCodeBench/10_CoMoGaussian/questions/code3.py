
class CoMoModule(nn.Module):
    def __init__(self,
                 num_views: int = 29,
                 view_dim: int = 32,
                 num_warp: int = 9,
                 method: str = 'euler',
                 adjoint: bool = False,
                 ) -> None:
        super(CoMoModule, self).__init__()

        self.num_warp = num_warp

        self.view_embedder = nn.Parameter(
            torch.zeros(num_views, view_dim).type(torch.float32), 
            requires_grad=True
        )

        self.linear_Rt = nn.ModuleList()
        
        self.encoder_rigid = nn.ModuleList()
        self.encoder_cmr = nn.ModuleList()
        
        self.diffeq_solver_rigid = nn.ModuleList()
        self.diffeq_solver_cmr = nn.ModuleList()
        
        self.decoder_rigid_w = nn.ModuleList()
        self.decoder_rigid_v = nn.ModuleList()
        self.decoder_rigid_theta = nn.ModuleList()
        
        self.decoder_cmr_rot = nn.ModuleList()
        self.decoder_cmr_trans = nn.ModuleList()

        self.mlpWeight = nn.ModuleList()
        self.mlpMask = nn.ModuleList()


        for i in range(num_views):
            
            self.linear_Rt.append(nn.Linear(12, view_dim))
            
            self.encoder_rigid.append(nn.Linear(view_dim + view_dim, view_dim))
            self.encoder_cmr.append(nn.Linear(view_dim + view_dim, view_dim))
            
            self.diffeq_solver_rigid.append(DiffEqSolver(
                odefunc=NeuralDerivative(view_dim=view_dim, num_views=num_views, num_warp=num_warp),
                method=method, num_warp=num_warp, adjoint=adjoint))
            
            self.diffeq_solver_cmr.append(DiffEqSolver(
                odefunc=NeuralDerivative(view_dim=view_dim, num_views=num_views, num_warp=num_warp),
                method=method, num_warp=num_warp, adjoint=adjoint))
            
            self.decoder_rigid_w.append(nn.Linear(view_dim // 2, 3))
            self.decoder_rigid_v.append(nn.Linear(view_dim // 2, 3))
            self.decoder_rigid_theta.append(nn.Linear(view_dim // 2, 1))
            
            self.decoder_cmr_rot.append(nn.Linear(view_dim // 2, 9))
            self.decoder_cmr_trans.append(nn.Linear(view_dim // 2, 3))
            
            gain = 0.00001 / (math.sqrt((view_dim // 2 + 3) / 6))
            self._init(self.decoder_rigid_w[i], gain=gain)
            self._init(self.decoder_rigid_v[i], gain=gain)
            self._init(self.decoder_rigid_theta[i], gain=gain)
            self._init(self.decoder_cmr_rot[i], gain=gain)
            self._init(self.decoder_cmr_trans[i], gain=gain)


        # conv, mlp_weight, mlp_mask from BAGS (https://github.com/snldmt/BAGS/)
        channels = 32
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=channels, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.InstanceNorm2d(channels),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.InstanceNorm2d(channels),
        )

        self.mlp_weight = nn.Conv2d(channels, 1, 1, bias=False)
        self.mlp_mask = nn.Conv2d(channels * num_warp, 1, 1, bias=False)
    
    
    def _init(self, 
              layer: nn.Module, 
              gain: float = 0.00001):
        
        nn.init.xavier_uniform_(layer.weight, gain=gain)
        if layer.bias is not None:
            layer.bias.data.fill_(0)


    def get_weight_and_mask(self, 
                            img: torch.Tensor = None,
                            idx_view: int = None,
                            ):
        
        feat = self.conv(img)
        weight = self.mlp_weight(feat)
        weight = F.softmax(weight, dim=0)
        
        feat_mask = rearrange(feat, 't c h w -> 1 (t c ) h w')
        mask = torch.sigmoid(self.mlp_mask(feat_mask))[0]

        return weight, mask
        
        
    def forward(self,
                Rt: torch.Tensor = None,
                idx_view: int = None,
                ) -> torch.Tensor:

        Rt_encoded = self.linear_Rt[idx_view](Rt[:3, :].reshape(-1))
        view_embed = self.view_embedder[idx_view]
        view_embed = torch.cat([view_embed, Rt_encoded], dim=-1)

        z_rigid = self.encoder_rigid[idx_view](view_embed)
        Z_rigid = self.diffeq_solver_rigid[idx_view](z_rigid)
        T_rigid = self.get_rigid_transformation(Z_rigid, idx_view)
        
        z_cmr = self.encoder_cmr[idx_view](view_embed)
        Z_cmr = self.diffeq_solver_cmr[idx_view](z_cmr)
        T_cmr, R_cmr = self.get_cmr_transformation(Z_cmr, idx_view)
        
        T_transform = torch.matmul(T_rigid, T_cmr)
        Rt_new = torch.einsum('ij, tjk -> tik', Rt, T_transform)
        
        w_loss = (torch.matmul(R_cmr, R_cmr.transpose(1, 2)) \
            - torch.eye(3)[None].repeat(self.num_warp, 1, 1).to(R_cmr)).abs().mean()
        
        return Rt_new, w_loss
    
    
    def get_rigid_transformation(self, 
                                 latent_rigid: torch.Tensor = None,
                                 idx_view: int = None
                                 ) -> torch.Tensor:
        
        ****EMPTY****
        
        return T_rigid

    def transform_SE3(self, 
                      exp_w_skew: torch.Tensor, 
                      p: torch.Tensor
                      ) -> torch.Tensor:
        
        delta_Rt = torch.cat([exp_w_skew, p], dim=-1)
        delta_Rt_fill = torch.tensor([0, 0, 0, 1])[None].repeat(delta_Rt.size(0), 1, 1).to(delta_Rt)
        delta_Rt = torch.cat([delta_Rt, delta_Rt_fill], dim=1)
        return delta_Rt
    
    def rodrigues_formula(self, 
                          w: torch.Tensor, 
                          theta: torch.Tensor,
                          ) -> torch.Tensor:
        
        term1 = torch.eye(3).to(w)
        term2 = torch.sin(theta) * w
        term3 = (1 - torch.cos(theta)) * torch.matmul(w, w)
        return term1 + term2 + term3
    
    def G_formula(self,
                  w: torch.Tensor, 
                  theta: torch.Tensor,
                  ) -> torch.Tensor:
        
        term1 = torch.eye(3)[None].to(w) * theta
        term2 = (1 - torch.cos(theta)) * w
        term3 = (theta - torch.sin(theta)) * torch.matmul(w, w)
        return term1 + term2 + term3

    def exp_map(self, 
                w: torch.Tensor,
                ) -> torch.Tensor:
        norm = torch.norm(w, dim=-1)[..., None] + 1e-10
        w = w / norm
        return w

    def skew_symmetric(self, 
                       w : torch.Tensor,
                       ) -> torch.Tensor:
        
        w1, w2, w3 = torch.chunk(w, 3, dim=-1)

        w_skew =  torch.cat([torch.zeros_like(w1), -w3, w2,
                             w3, torch.zeros_like(w1), -w1,
                             -w2, w1, torch.zeros_like(w1)], dim=-1)
        w_skew = w_skew.reshape(-1, 3, 3)
        return w_skew
    
    def get_cmr_transformation(self, 
                               latent_cmr: torch.Tensor = None,
                               idx_view: int = None
                               ) -> torch.Tensor:
        
        ****EMPTY****
        
        return T_cmr, R_cmr
